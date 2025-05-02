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
    expanded: {
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
    expanded: true,
  },
};

export const HiddenContent: Story = {
  args: {
    expanded: false,
  },
};
