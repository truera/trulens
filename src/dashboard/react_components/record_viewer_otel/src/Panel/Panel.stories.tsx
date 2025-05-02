import type { Meta, StoryObj } from '@storybook/react';
import Panel from './Panel';

type Story = StoryObj<typeof Panel>;

const meta: Meta<typeof Panel> = {
  title: 'Components/Panel',
  component: Panel,
  args: {
    header: 'Panel Header',
    children: 'Panel Content',
  },
  argTypes: {
    defaultExpanded: {
      control: false,
      table: {
        disable: true
      }
    }
  }
};

export default meta;

export const ExpandedContent: Story = {
  args: {
    defaultExpanded: true,
  },
};

export const HiddenContent: Story = {
  args: {
    defaultExpanded: false,
  },
};
