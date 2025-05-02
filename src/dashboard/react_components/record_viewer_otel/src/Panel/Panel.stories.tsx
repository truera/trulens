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
};

export default meta;

export const Default: Story = {};
